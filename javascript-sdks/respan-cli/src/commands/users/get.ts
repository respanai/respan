import { Args } from '@oclif/core';
import { BaseCommand } from '../../lib/base-command.js';

export default class UsersGet extends BaseCommand {
  static description = 'Get a specific user (customer)';
  static args = { id: Args.string({ description: 'Customer identifier', required: true }) };
  static flags = { ...BaseCommand.baseFlags };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(UsersGet);
    this.globalFlags = flags;
    try {
      const client = this.getClient();
      const data = await this.spin('Fetching user', () =>
        client.users.retrieveUser({ Authorization: this.getAuthHeader(), customer_identifier: args.id }),
      );
      this.log(JSON.stringify(data, null, 2));
    } catch (error) {
      this.handleError(error);
    }
  }
}
